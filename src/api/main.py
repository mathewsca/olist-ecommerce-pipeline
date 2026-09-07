"""
FabricaIA - FastAPI Application

This module provides a REST API for model inference using FabricaIA trained models.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

# Import FabricaIA modules
from src.data.processor import DataProcessor
from src.models.trainer import ModelTrainer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="FabricaIA API",
    description="Machine Learning API using FabricaIA framework",
    version="1.0.0",
)

# Global variables
models = {}
config = None
data_processor = DataProcessor()
model_trainer = ModelTrainer()


class PredictionRequest(BaseModel):
    """Request model for predictions."""

    features: Dict[str, Any]
    model_name: str = "random_forest"


class PredictionResponse(BaseModel):
    """Response model for predictions."""

    prediction: Any
    probability: Optional[Dict[str, float]] = None
    model_name: str
    confidence: Optional[float] = None


class ModelInfo(BaseModel):
    """Model information response."""

    model_name: str
    model_type: str
    features: List[str]
    accuracy: Optional[float] = None


@app.on_event("startup")
async def startup_event():
    """Load models and configuration on startup."""
    global models, config

    try:
        # Load configuration
        with open("config/config.yaml", "r") as file:
            config = yaml.safe_load(file)

        # Load available models
        model_dir = Path(config["MODEL_PATHS"]["trained"])
        if model_dir.exists():
            for model_file in model_dir.glob("*.pkl"):
                model_name = model_file.stem.replace("_model", "")
                models[model_name] = model_trainer.load_model(str(model_file))
                logger.info(f"Loaded model: {model_name}")

        logger.info(f"API started successfully. Loaded {len(models)} models.")

    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "FabricaIA API",
        "version": "1.0.0",
        "available_models": list(models.keys()),
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "models_loaded": len(models)}


@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List all available models."""
    model_info = []

    for model_name, model in models.items():
        info = ModelInfo(
            model_name=model_name,
            model_type=type(model).__name__,
            features=getattr(model, "feature_names_in_", []).tolist()
            if hasattr(model, "feature_names_in_")
            else [],
        )
        model_info.append(info)

    return model_info


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make predictions using a trained model."""
    try:
        # Check if model exists
        if request.model_name not in models:
            raise HTTPException(
                status_code=404, detail=f"Model '{request.model_name}' not found"
            )

        model = models[request.model_name]

        # Convert features to DataFrame
        features_df = pd.DataFrame([request.features])

        # Preprocess features if needed
        # This would depend on your specific preprocessing requirements
        processed_features = features_df

        # Make prediction
        prediction = model_trainer.predict(model, processed_features)

        # Get probabilities if available
        probability = None
        if hasattr(model, "predict_proba"):
            proba = model_trainer.predict_proba(model, processed_features)
            classes = getattr(model, "classes_", None)
            if classes is not None:
                probability = {
                    str(classes[i]): float(proba[0][i]) for i in range(len(classes))
                }

        # Calculate confidence (max probability for classification)
        confidence = None
        if probability:
            confidence = max(probability.values())

        return PredictionResponse(
            prediction=prediction[0].item() if len(prediction) == 1 else prediction.tolist(),
            probability=probability,
            model_name=request.model_name,
            confidence=confidence,
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch")
async def predict_batch(
    file: UploadFile = File(...), model_name: str = "random_forest"
):
    """Make batch predictions from CSV file."""
    try:
        # Check if model exists
        if model_name not in models:
            raise HTTPException(
                status_code=404, detail=f"Model '{model_name}' not found"
            )

        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(pd.io.common.StringIO(contents.decode("utf-8")))

        model = models[model_name]

        # Make predictions
        predictions = model_trainer.predict(model, df)

        # Add predictions to DataFrame
        df["prediction"] = predictions

        # Convert to JSON
        result = df.to_dict("records")

        return {
            "model_name": model_name,
            "predictions": result,
            "total_samples": len(df),
        }

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/{model_name}/info")
async def get_model_info(model_name: str):
    """Get detailed information about a specific model."""
    if model_name not in models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    model = models[model_name]

    info: Dict[str, Any] = {
        "model_name": model_name,
        "model_type": type(model).__name__,
        "parameters": getattr(model, "get_params", lambda: {})(),
    }

    # Add feature information if available
    if hasattr(model, "feature_names_in_"):
        info["features"] = model.feature_names_in_.tolist()

    # Add classes information for classification models
    if hasattr(model, "classes_"):
        info["classes"] = model.classes_.tolist()

    return info


@app.post("/model/{model_name}/retrain")
async def retrain_model(model_name: str, data_path: str):
    """Retrain a model with new data."""
    try:
        # This would implement retraining logic
        # For now, just return a placeholder response
        return {
            "message": f"Retraining {model_name} with data from {data_path}",
            "status": "initiated",
        }

    except Exception as e:
        logger.error(f"Retraining error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
