"""
FabricaIA - Airflow DAG Example

This DAG demonstrates how to use FabricaIA modules in an Airflow workflow
for automated machine learning pipelines.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Variable
try:
    from airflow.providers.standard.operators.bash import BashOperator
    from airflow.providers.standard.operators.python import PythonOperator
except ImportError:
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
    schedule=timedelta(days=1),
    catchup=False,
    tags=["ml", "fabricaia", "pipeline"],
)


def _ensure_project_path():
    import sys
    from pathlib import Path
    project_root = str(Path(__file__).resolve().parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    if "/opt/airflow" not in sys.path:
        sys.path.append("/opt/airflow")


def _resolve_data_path() -> str:
    from pathlib import Path
    custom = Variable.get("data_path", default_var=None)
    if custom and Path(custom).exists():
        return custom
    for p in ["data/raw/obras_publicas.csv", "data/raw/sample_data.csv"]:
        if Path(p).exists():
            return p
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    sample_path = "data/raw/obras_publicas.csv"
    try:
        from scripts.generate_dataset import generate_obras_dataset
        generate_obras_dataset(sample_path, n_samples=300)
    except Exception:
        import numpy as np
        import pandas as pd
        df = pd.DataFrame({
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "target": np.random.randint(0, 2, 100),
        })
        df.to_csv(sample_path, index=False)
    return sample_path


def _resolve_target_column(df) -> str:
    target_var = Variable.get("target_column", default_var=None)
    if target_var and target_var in df.columns:
        return target_var
    if "atraso_risco" in df.columns:
        return "atraso_risco"
    if "target" in df.columns:
        return "target"
    return df.columns[-1]


def load_and_explore_data(**context):
    """Load and explore data using FabricaIA modules."""
    _ensure_project_path()
    from pathlib import Path
    from src.data.processor import DataProcessor
    from src.visualization.plots import DataVisualizer

    data_path = _resolve_data_path()
    data_processor = DataProcessor()
    visualizer = DataVisualizer()

    df = data_processor.load_data(data_path)
    Path("logs").mkdir(parents=True, exist_ok=True)
    visualizer.plot_missing_data(df, "logs/missing_data.png")

    return {
        "data_path": data_path,
        "data_shape": df.shape,
        "columns": df.columns.tolist(),
        "missing_values": df.isnull().sum().to_dict(),
    }


def check_data_drift(**context):
    """Detect statistical drift between incoming data and reference baseline."""
    _ensure_project_path()
    from pathlib import Path
    import numpy as np
    import pandas as pd

    data_path = _resolve_data_path()
    if not Path(data_path).exists():
        return {"drift_detected": False, "reason": "data_path not found"}

    df = pd.read_csv(data_path)
    half = len(df) // 2
    ref_batch = df.iloc[:half]
    cur_batch = df.iloc[half:]

    drift_report = {}
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols[:4]:
        ref_mean = float(ref_batch[col].mean())
        cur_mean = float(cur_batch[col].mean())
        std_val = ref_batch[col].std()
        shift = abs(cur_mean - ref_mean) / (std_val + 1e-6) if pd.notnull(std_val) else 0.0
        drift_report[col] = {
            "ref_mean": round(ref_mean, 2),
            "cur_mean": round(cur_mean, 2),
            "normalized_shift": round(shift, 3),
        }

    drift_detected = any(v["normalized_shift"] > 0.5 for v in drift_report.values())
    return {"drift_detected": drift_detected, "report": drift_report}


def preprocess_data(**context):
    """Preprocess data using FabricaIA modules."""
    _ensure_project_path()
    from pathlib import Path
    from src.data.processor import DataProcessor

    data_path = _resolve_data_path()
    data_processor = DataProcessor()

    df = data_processor.load_data(data_path)
    target_column = _resolve_target_column(df)
    df_clean = data_processor.clean_data(
        df, drop_duplicates=True, handle_missing="fill"
    )
    df_encoded = data_processor.encode_categorical(df_clean, method="label")
    df_scaled = data_processor.scale_features(df_encoded, exclude_columns=[target_column])

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df_scaled.to_csv("data/processed/preprocessed_data.csv", index=False)

    return {"processed_shape": df_scaled.shape}


def engineer_features(**context):
    """Engineer features using FabricaIA modules."""
    _ensure_project_path()
    from pathlib import Path
    import pandas as pd
    from src.features.engineering import FeatureEngineer

    df = pd.read_csv("data/processed/preprocessed_data.csv")
    feature_engineer = FeatureEngineer()

    numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if len(numerical_cols) >= 2:
        df_poly = feature_engineer.create_polynomial_features(df, numerical_cols[:2])
        df_interact = feature_engineer.create_interaction_features(
            df_poly, [(numerical_cols[0], numerical_cols[1])]
        )
    else:
        df_interact = df

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df_interact.to_csv("data/processed/engineered_data.csv", index=False)

    return {"engineered_shape": df_interact.shape}


def sync_feature_store(**context):
    """Synchronize processed features to feature store (Parquet format)."""
    _ensure_project_path()
    from pathlib import Path
    import pandas as pd

    engineered_path = "data/processed/engineered_data.csv"
    if Path(engineered_path).exists():
        df = pd.read_csv(engineered_path)
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        feature_store_path = "data/processed/feature_store.parquet"
        df.to_parquet(feature_store_path, index=False)
        return {"feature_store_records": len(df), "path": feature_store_path}
    return {"status": "skipped"}


def train_models(**context):
    """Train models using FabricaIA modules."""
    _ensure_project_path()
    from pathlib import Path
    import pandas as pd
    from src.data.processor import DataProcessor
    from src.models.trainer import ModelTrainer
    from src.visualization.plots import DataVisualizer

    df = pd.read_csv("data/processed/engineered_data.csv")
    target_column = _resolve_target_column(df)

    data_processor = DataProcessor()
    model_trainer = ModelTrainer()
    visualizer = DataVisualizer()

    X_train, X_test, y_train, y_test = data_processor.split_data(
        df, target_column, test_size=0.2, random_state=42
    )

    algorithms = ["random_forest", "logistic_regression"]
    results = {}

    Path("models/trained").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)

    for algorithm in algorithms:
        if model_trainer.mlflow_tracker:
            model_trainer.mlflow_tracker.start_run(run_name=f"{algorithm}_run")
        try:
            model = model_trainer.train_model(X_train, y_train, algorithm)
            metrics = model_trainer.evaluate_model(model, X_test, y_test)

            model_path = f"models/trained/{algorithm}_model.pkl"
            model_trainer.save_model(model, model_path)

            if hasattr(model, "feature_importances_"):
                importance_df = model_trainer.get_feature_importance(model)
                visualizer.plot_feature_importance(
                    importance_df, save_path=f"logs/{algorithm}_feature_importance.png"
                )

            results[algorithm] = metrics
        finally:
            if model_trainer.mlflow_tracker:
                model_trainer.mlflow_tracker.end_run()

    return results


def evaluate_models(**context):
    """Evaluate and compare model performance."""
    _ensure_project_path()

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


def promote_production_model(**context):
    """Quality gate: promote best model to production if metric threshold is met."""
    _ensure_project_path()

    import shutil
    from pathlib import Path

    eval_results = context["task_instance"].xcom_pull(task_ids="evaluate_models")
    best_model = eval_results.get("best_model", "random_forest")
    all_results = eval_results.get("all_results", {})
    metrics = all_results.get(best_model, {})

    score = metrics.get("accuracy", metrics.get("r2_score", 0))
    threshold = float(Variable.get("production_threshold", default_var="0.75"))

    candidate_path = Path(f"models/trained/{best_model}_model.pkl")
    prod_path = Path("models/trained/production_model.pkl")

    if score >= threshold and candidate_path.exists():
        shutil.copy(candidate_path, prod_path)
        decision = f"PROMOTED: {best_model} (score: {score:.4f} >= threshold: {threshold:.4f})"
        status = "promoted"
    else:
        decision = f"RETAINED: current production model kept (score: {score:.4f} < threshold: {threshold:.4f})"
        status = "rejected"

    return {"status": status, "decision": decision, "production_model": str(prod_path)}


# Task definitions
load_data_task = PythonOperator(
    task_id="load_and_explore_data",
    python_callable=load_and_explore_data,
    dag=dag,
)

drift_check_task = PythonOperator(
    task_id="check_data_drift",
    python_callable=check_data_drift,
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

feature_store_task = PythonOperator(
    task_id="sync_feature_store",
    python_callable=sync_feature_store,
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

promote_model_task = PythonOperator(
    task_id="promote_production_model",
    python_callable=promote_production_model,
    dag=dag,
)

# Cleanup task
cleanup_task = BashOperator(
    task_id="cleanup_temp_files",
    bash_command='find /tmp -name "*.tmp" -user $(whoami) -delete 2>/dev/null || true',
    dag=dag,
)

# Task dependencies
(
    load_data_task
    >> drift_check_task
    >> preprocess_task
    >> engineer_features_task
    >> feature_store_task
    >> train_models_task
    >> evaluate_models_task
    >> promote_model_task
    >> cleanup_task
)
