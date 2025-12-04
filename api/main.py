"""
FastAPI REST API for ML Model Inference.

Provides endpoints for:
- Health check
- Model information
- Single predictions
- Batch predictions

TODO: Customize based on your project requirements.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.inference import (
    predict_classification,
    predict_regression,
    batch_predict,
    get_model_info,
    load_model,
)

# ============================================================================
# APP INITIALIZATION
# ============================================================================

app = FastAPI(
    title="ML Model API",
    description="REST API for machine learning model inference",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class FeatureInput(BaseModel):
    """Input features for prediction."""
    features: Union[Dict[str, float], List[float]] = Field(
        ...,
        description="Feature values as dict or list",
        example={"feature_1": 1.0, "feature_2": 2.0}
    )


class BatchFeatureInput(BaseModel):
    """Batch input features for multiple predictions."""
    features_list: List[Union[Dict[str, float], List[float]]] = Field(
        ...,
        description="List of feature dicts or lists",
        example=[
            {"feature_1": 1.0, "feature_2": 2.0},
            {"feature_1": 3.0, "feature_2": 4.0}
        ]
    )


class PredictionResponse(BaseModel):
    """Response for a single prediction."""
    prediction: Union[int, float]
    probabilities: Optional[List[float]] = None
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""
    predictions: List[Union[int, float]]
    count: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    message: str


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_name: Optional[str] = None
    model_type: Optional[str] = None
    n_features: Optional[int] = None
    feature_names: Optional[List[str]] = None
    metrics: Optional[Dict[str, float]] = None


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check."""
    return {
        "status": "ok",
        "message": "ML Model API is running. Visit /docs for API documentation."
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Try to load model to verify it's available
        load_model()
        return {
            "status": "healthy",
            "message": "Model loaded successfully"
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model not available: {str(e)}"
        )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Get information about the loaded model."""
    try:
        info = get_model_info()
        return info
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting model info: {str(e)}"
        )


@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: FeatureInput):
    """
    Make a single prediction.
    
    Accepts feature values as a dictionary or list.
    Returns prediction with optional probabilities and confidence.
    """
    try:
        # Get model type
        metadata = get_model_info()
        model_type = metadata.get("model_type", "classification")
        
        if model_type == "classification":
            result = predict_classification(
                input_data.features,
                return_proba=True
            )
            return result
        
        elif model_type == "regression":
            prediction = predict_regression(input_data.features)
            return {
                "prediction": prediction,
                "metadata": metadata
            }
        
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported model type: {model_type}"
            )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(input_data: BatchFeatureInput):
    """
    Make predictions for multiple samples.
    
    Accepts a list of feature dicts or lists.
    Returns a list of predictions.
    """
    try:
        predictions = batch_predict(input_data.features_list)
        
        return {
            "predictions": predictions,
            "count": len(predictions)
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction error: {str(e)}"
        )


# ============================================================================
# STARTUP EVENT
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    try:
        load_model()
        print("✓ Model loaded successfully on startup")
    except Exception as e:
        print(f"⚠️ Warning: Could not load model on startup: {e}")
        print("   Model will be loaded on first prediction request")


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
