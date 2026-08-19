# Quick Start

## Setup

### Prerequisites
- Python 3.8+
- Node.js 16+
- Ports 8000 and 5173 available

### Backend

Terminal 1:
```bash
cd backend
pip install -r requirements.txt          # First time only
python train_model.py                    # First time only
python -m uvicorn app:app --reload
```

API: http://localhost:8000
Docs: http://localhost:8000/docs

### Frontend

Terminal 2:
```bash
cd frontend
npm install                              # First time only
npm run dev
```

App: http://localhost:5173

## Test API

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

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Info |
| GET | `/health` | Status |
| GET | `/model-info` | Metrics |
| GET | `/feature-ranges` | Input ranges |
| POST | `/predict` | Single prediction |
| POST | `/predict-bulk` | Batch predictions |

## Troubleshooting

**Port in use:**
```bash
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

**Missing dependencies:**
```bash
cd backend
pip install -r requirements.txt
```

**npm issues:**
```bash
cd frontend
rm -rf node_modules
npm install
```

**Model not found:**
```bash
cd backend
python train_model.py              # Generate model artifacts
```

## Model

- R²: 0.84
- MAE: $30k
- Training: ~10 seconds
- Prediction: <100ms

- Ensure backend is fully restarted after code changes
- Model must be trained: `python train_model.py`

## 📝 Model Training Details

Training automatically happens once when you run:
```bash
python train_model.py
```

This creates:
- `models/xgboost_model.pkl` - Trained model
- `models/feature_scaler.pkl` - Feature normalizer
- `models/training_metrics.pkl` - Performance stats

To retrain (e.g., after code changes):
```bash
python train_model.py
# Server will auto-reload if using --reload flag
```

## 🎨 UI Features

**Model Stats Panel**
- R² Score, RMSE, MAE displays
- Performance chart
- Training information

**Prediction Form**
- 8 property input fields
- Input ranges validation
- Helpful tooltips
- Loading indicator

**Results Panel**
- Large price display
- Confidence metrics
- Input review
- Limitations disclaimer

## 💡 Tips

1. **First load slower**: Model loads on server startup (~1 second)
2. **API Documentation**: Visit http://localhost:8000/docs for Swagger UI
3. **Feature tooltips**: Hover over `?` icons in form for explanations
4. **Range validation**: Each field shows acceptable range
5. **Price format**: Predictions displayed in USD with comma separators

## 🚀 Next Steps

1. **Test locally**: Start both servers and try a prediction
2. **Explore API**: Visit `/docs` for interactive testing
3. **Read docs**: Check README.md for full documentation
4. **Review submission**: Read SUBMISSION.md for complete details

## Questions?

Refer to:
- `README.md` - Full documentation
- `SUBMISSION.md` - Architecture & decisions
- `http://localhost:8000/docs` - Interactive API docs
- Code comments - Inline explanations

Happy testing! 🎉
