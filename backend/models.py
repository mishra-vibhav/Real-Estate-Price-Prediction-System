from pydantic import BaseModel, Field
from typing import List, Optional, Dict


class PredictionRequest(BaseModel):
    MedInc: float = Field(..., description="Median income", example=8.3)
    HouseAge: float = Field(..., description="House age", example=41.0)
    AveRooms: float = Field(..., description="Avg rooms", example=6.98)
    AveBedrms: float = Field(..., description="Avg bedrooms", example=1.02)
    Population: float = Field(..., description="Population", example=322.0)
    AveOccup: float = Field(..., description="Avg occupancy", example=2.55)
    Latitude: float = Field(..., description="Latitude", example=37.88)
    Longitude: float = Field(..., description="Longitude", example=-122.23)
    
    class Config:
        json_schema_extra = {
            "example": {
                "MedInc": 8.3,
                "HouseAge": 41.0,
                "AveRooms": 6.98,
                "AveBedrms": 1.02,
                "Population": 322.0,
                "AveOccup": 2.55,
                "Latitude": 37.88,
                "Longitude": -122.23,
            }
        }


class PredictionResponse(BaseModel):
    predicted_price: float
    confidence: Dict[str, float]
    input_features: Dict[str, float]


class FeatureStatistics(BaseModel):
    mean: float
    std: float
    min: float
    max: float


class ModelInfo(BaseModel):
    model_type: str
    features: List[str]
    r2_score: float
    rmse: float
    mae: float
    training_samples: int
    cv_r2_mean: float
    cv_r2_std: float


class BulkPredictionRequest(BaseModel):
    data: List[PredictionRequest]


class BulkPredictionResponse(BaseModel):
    predictions: List[Dict]
    count: int
