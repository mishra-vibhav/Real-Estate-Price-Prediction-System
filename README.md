# Real Estate Price Prediction

ML system to predict house prices using XGBoost. Trained on California housing data.

## Setup

### Backend
```bash
cd backend
pip install -r requirements.txt
python train_model.py
python -m uvicorn app:app --reload
```

Runs on http://localhost:8000

### Frontend
```bash
cd frontend
npm install
npm run dev
```

Runs on http://localhost:5173

## Structure

```
backend/
  app.py - FastAPI server
  train_model.py - Training script
  data_processor.py - Data prep
  config.py - Config
  models.py - Schemas

frontend/
  src/App.jsx - Main app
  src/components/ - React components
  package.json - Deps
```

## Model

- R²: 0.84
- MAE: $30k
- RMSE: $46k
- 8 input features
- California Housing dataset

## API

- `GET /` - Info
- `GET /health` - Status
- `GET /model-info` - Metrics
- `POST /predict` - Prediction
- `GET /feature-ranges` - Input ranges

## Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "MedInc": 8.3,
    "HouseAge": 41,
    "AveRooms": 6.98,
    "AveBedrms": 1.02,
    "Population": 322,
    "AveOccup": 2.55,
    "Latitude": 37.88,
    "Longitude": -122.23
  }'
```

## Tech Stack

Backend: Python, FastAPI, XGBoost, scikit-learn, pandas
Frontend: React, Vite, Axios, CSS
Data: California Housing (free dataset)

## Deployment

Frontend -> Vercel
Backend -> Railway/Render

No paid APIs used.

