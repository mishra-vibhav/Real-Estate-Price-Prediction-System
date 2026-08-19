import React from 'react';
import './ResultsDisplay.css';

function ResultsDisplay({ prediction }) {
  const price = prediction.predicted_price;
  const formatted_price = new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 0,
  }).format(price);

  return (
    <div className="results-display">
      <h2>Prediction Result</h2>

      <div className="price-box">
        <div className="price-label">Estimated Property Value</div>
        <div className="price-value">{formatted_price}</div>
      </div>

      <div className="confidence-section">
        <h3>Model Confidence</h3>
        <div className="confidence-metrics">
          <div className="metric">
            <span className="metric-label">R² Score</span>
            <span className="metric-value">
              {(prediction.confidence.r2_score * 100).toFixed(2)}%
            </span>
          </div>
          <div className="metric">
            <span className="metric-label">Typical Error (MAE)</span>
            <span className="metric-value">
              ${(prediction.confidence.model_mae / 1000).toFixed(1)}K
            </span>
          </div>
        </div>
      </div>

      <div className="inputs-section">
        <h3>Input Features Used</h3>
        <div className="inputs-grid">
          {Object.entries(prediction.input_features).map(([key, value]) => (
            <div key={key} className="input-item">
              <span className="input-label">{key.replace(/_/g, ' ')}</span>
              <span className="input-value">
                {typeof value === 'number' ? value.toFixed(2) : value}
              </span>
            </div>
          ))}
        </div>
      </div>

      <div className="info-box">
        <p>
          💡 This prediction is based on California housing market data.
          Actual market prices may vary based on additional factors not captured by the model.
        </p>
      </div>
    </div>
  );
}

export default ResultsDisplay;
