from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pickle
from pathlib import Path
import numpy as np
import logging
from config import MODELS_DIR, API_TITLE, API_VERSION
from models import (
    PredictionRequest, PredictionResponse, ModelInfo,
    BulkPredictionRequest, BulkPredictionResponse
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="Real estate price prediction"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
scaler = None
feature_names = None
training_metrics = None


def load_model_artifacts():
    global model, scaler, feature_names, training_metrics
    
    try:
        model_path = MODELS_DIR / "xgboost_model.pkl"
        scaler_path = MODELS_DIR / "feature_scaler.pkl"
        features_path = MODELS_DIR / "feature_names.pkl"
        metrics_path = MODELS_DIR / "training_metrics.pkl"
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        with open(features_path, 'rb') as f:
            feature_names = pickle.load(f)
        
        with open(metrics_path, 'rb') as f:
            training_metrics = pickle.load(f)
        
        logger.info("Model artifacts loaded successfully")
    except FileNotFoundError as e:
        logger.error(f"Model artifacts not found: {e}")
        logger.error("Please run: python train_model.py")
        raise RuntimeError("Model not trained. Run train_model.py first.")


@app.on_event("startup")
async def startup_event():
    load_model_artifacts()


@app.get("/", tags=["Health"])
async def root():
    return {
        "name": "Real Estate Price Prediction API",
        "version": API_VERSION,
        "status": "running",
        "endpoints": {
            "predict": "/predict",
            "predict_bulk": "/predict-bulk",
            "model_info": "/model-info",
            "health": "/health"
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.get("/model-info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfo(
        model_type="XGBoost Regressor",
        features=training_metrics['features'],
        r2_score=training_metrics['r2_score'],
        rmse=training_metrics['rmse'],
        mae=training_metrics['mae'],
        training_samples=training_metrics['train_size'],
        cv_r2_mean=training_metrics['cv_r2_mean'],
        cv_r2_std=training_metrics['cv_r2_std'],
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_price(request: PredictionRequest):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        import pandas as pd
        
        # Create features in exact order that matches training
        features_df = pd.DataFrame({
            'MedInc': [request.MedInc],
            'HouseAge': [request.HouseAge],
            'AveRooms': [request.AveRooms],
            'AveBedrms': [request.AveBedrms],
            'Population': [request.Population],
            'AveOccup': [request.AveOccup],
            'Latitude': [request.Latitude],
            'Longitude': [request.Longitude],
        })
        
        # Feature engineering
        features_df['rooms_per_household'] = features_df['AveRooms']
        features_df['bedrooms_per_household'] = features_df['AveBedrms']
        features_df['population_per_household'] = features_df['Population']
        features_df['coastal_proximity'] = (features_df['Latitude'] > 37.5).astype(int).values
        
        feature_order = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude',
                        'rooms_per_household', 'bedrooms_per_household', 'population_per_household', 'coastal_proximity']
        features_df = features_df[feature_order]
        
        features_scaled = scaler.transform(features_df)
        prediction = model.predict(features_scaled)[0]
        
        return PredictionResponse(
            predicted_price=float(prediction),
            confidence={
                "r2_score": training_metrics['r2_score'],
                "rmse": training_metrics['rmse'],
                "model_mae": training_metrics['mae'],
            },
            input_features={
                "median_income": request.MedInc,
                "house_age": request.HouseAge,
                "avg_rooms": request.AveRooms,
                "avg_bedrooms": request.AveBedrms,
                "population": request.Population,
                "avg_occupancy": request.AveOccup,
                "latitude": request.Latitude,
                "longitude": request.Longitude,
            }
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.post("/predict-bulk", response_model=BulkPredictionResponse, tags=["Predictions"])
async def predict_bulk(request: BulkPredictionRequest):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        import pandas as pd
        
        predictions = []
        
        for item in request.data:
            features_df = pd.DataFrame({
                'MedInc': [item.MedInc],
                'HouseAge': [item.HouseAge],
                'AveRooms': [item.AveRooms],
                'AveBedrms': [item.AveBedrms],
                'Population': [item.Population],
                'AveOccup': [item.AveOccup],
                'Latitude': [item.Latitude],
                'Longitude': [item.Longitude],
            })
            
            features_df['rooms_per_household'] = features_df['AveRooms']
            features_df['bedrooms_per_household'] = features_df['AveBedrms']
            features_df['population_per_household'] = features_df['Population']
            features_df['coastal_proximity'] = (features_df['Latitude'] > 37.5).astype(int).values
            
            feature_order = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude',
                            'rooms_per_household', 'bedrooms_per_household', 'population_per_household', 'coastal_proximity']
            features_df = features_df[feature_order]
            
            features_scaled = scaler.transform(features_df)
            prediction = model.predict(features_scaled)[0]
            
            predictions.append({
                "predicted_price": float(prediction),
                "input_features": {
                    "median_income": item.MedInc,
                    "house_age": item.HouseAge,
                    "avg_rooms": item.AveRooms,
                }
            })
        
        return BulkPredictionResponse(
            predictions=predictions,
            count=len(predictions)
        )
    
    except Exception as e:
        logger.error(f"Bulk prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Bulk prediction failed: {str(e)}")


@app.get("/feature-ranges", tags=["Model"])
async def get_feature_ranges():
    if training_metrics is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "MedInc": {"min": 0.5, "max": 15.0},
        "HouseAge": {"min": 1, "max": 52},
        "AveRooms": {"min": 1, "max": 12},
        "AveBedrms": {"min": 0.5, "max": 5},
        "Population": {"min": 3, "max": 35000},
        "AveOccup": {"min": 0.5, "max": 10},
        "Latitude": {"min": 32.54, "max": 41.95},
        "Longitude": {"min": -124.35, "max": -114.13},
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
