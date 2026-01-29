"""
FabricaIA - Airflow DAG Example

This DAG demonstrates how to use FabricaIA modules in an Airflow workflow
for automated machine learning pipelines.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

# Default arguments
default_args = {
    "owner": "fabricaia",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# DAG definition
dag = DAG(
    "fabricaia_ml_pipeline",
    default_args=default_args,
    description="FabricaIA Machine Learning Pipeline",
    schedule_interval=timedelta(days=1),
    catchup=False,
    tags=["ml", "fabricaia", "pipeline"],
)


def load_and_explore_data(**context):
    """Load and explore data using FabricaIA modules."""
    import sys

    sys.path.append("/opt/airflow")

    from src.data.processor import DataProcessor
    from src.visualization.plots import DataVisualizer

    # Get data path from Airflow variables
    data_path = Variable.get("data_path", default_var="data/raw/sample_data.csv")

    # Initialize processors
    data_processor = DataProcessor()
    visualizer = DataVisualizer()

    # Load data
    df = data_processor.load_data(data_path)

    # Explore data
    visualizer.plot_missing_data(df, "logs/missing_data.png")

    # Store data info in XCom
    return {
        "data_shape": df.shape,
        "columns": df.columns.tolist(),
        "missing_values": df.isnull().sum().to_dict(),
    }


def preprocess_data(**context):
    """Preprocess data using FabricaIA modules."""
    import sys

    sys.path.append("/opt/airflow")

    from src.data.processor import DataProcessor

    # Get data path
    data_path = Variable.get("data_path", default_var="data/raw/sample_data.csv")

    # Initialize processor
    data_processor = DataProcessor()

    # Load and preprocess data
    df = data_processor.load_data(data_path)
    df_clean = data_processor.clean_data(
        df, drop_duplicates=True, handle_missing="fill"
    )
    df_encoded = data_processor.encode_categorical(df_clean, method="label")
    df_scaled = data_processor.scale_features(df_encoded)

    # Save processed data
    df_scaled.to_csv("data/processed/preprocessed_data.csv", index=False)

    return {"processed_shape": df_scaled.shape}


def engineer_features(**context):
    """Engineer features using FabricaIA modules."""
    import sys

    sys.path.append("/opt/airflow")

    import pandas as pd

    from src.features.engineering import FeatureEngineer

    # Load processed data
    df = pd.read_csv("data/processed/preprocessed_data.csv")

    # Initialize feature engineer
    feature_engineer = FeatureEngineer()

    # Engineer features
    numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if len(numerical_cols) >= 2:
        df_poly = feature_engineer.create_polynomial_features(df, numerical_cols[:2])
        df_interact = feature_engineer.create_interaction_features(
            df_poly, [(numerical_cols[0], numerical_cols[1])]
        )
    else:
        df_interact = df

    # Save engineered data
    df_interact.to_csv("data/processed/engineered_data.csv", index=False)

    return {"engineered_shape": df_interact.shape}


def train_models(**context):
    """Train models using FabricaIA modules."""
    import sys

    sys.path.append("/opt/airflow")

    import pandas as pd

    from src.data.processor import DataProcessor
    from src.models.trainer import ModelTrainer
    from src.visualization.plots import DataVisualizer

    # Get target column from Airflow variables
    target_column = Variable.get("target_column", default_var="target")

    # Load engineered data
    df = pd.read_csv("data/processed/engineered_data.csv")

    # Initialize processors
    data_processor = DataProcessor()
    model_trainer = ModelTrainer()
    visualizer = DataVisualizer()

    # Split data
    X_train, X_test, y_train, y_test = data_processor.split_data(
        df, target_column, test_size=0.2, random_state=42
    )

    # Train models
    algorithms = ["random_forest", "logistic_regression"]
    results = {}

    for algorithm in algorithms:
        model = model_trainer.train_model(X_train, y_train, algorithm)
        metrics = model_trainer.evaluate_model(model, X_test, y_test)

        # Save model
        model_path = f"models/trained/{algorithm}_model.pkl"
        model_trainer.save_model(model, model_path)

        # Plot feature importance if available
        if hasattr(model, "feature_importances_"):
            importance_df = model_trainer.get_feature_importance(model)
            visualizer.plot_feature_importance(
                importance_df, save_path=f"logs/{algorithm}_feature_importance.png"
            )

        results[algorithm] = metrics

    return results


def evaluate_models(**context):
    """Evaluate and compare model performance."""
    import sys

    sys.path.append("/opt/airflow")

    from src.visualization.plots import DataVisualizer

    # Get results from previous task
    results = context["task_instance"].xcom_pull(task_ids="train_models")

    # Initialize visualizer
    visualizer = DataVisualizer()

    # Create comparison plots
    for algorithm, metrics in results.items():
        visualizer.plot_model_performance(
            metrics, algorithm, f"logs/{algorithm}_performance.png"
        )

    # Log best model
    best_model = max(
        results.keys(),
        key=lambda x: results[x].get("accuracy", results[x].get("r2_score", 0)),
    )

    return {"best_model": best_model, "all_results": results}


# Task definitions
load_data_task = PythonOperator(
    task_id="load_and_explore_data",
    python_callable=load_and_explore_data,
    dag=dag,
)

preprocess_task = PythonOperator(
    task_id="preprocess_data",
    python_callable=preprocess_data,
    dag=dag,
)

engineer_features_task = PythonOperator(
    task_id="engineer_features",
    python_callable=engineer_features,
    dag=dag,
)

train_models_task = PythonOperator(
    task_id="train_models",
    python_callable=train_models,
    dag=dag,
)

evaluate_models_task = PythonOperator(
    task_id="evaluate_models",
    python_callable=evaluate_models,
    dag=dag,
)

# Cleanup task
cleanup_task = BashOperator(
    task_id="cleanup_temp_files",
    bash_command='find /tmp -name "*.tmp" -delete',
    dag=dag,
)

# Task dependencies
(
    load_data_task
    >> preprocess_task
    >> engineer_features_task
    >> train_models_task
    >> evaluate_models_task
    >> cleanup_task
)
