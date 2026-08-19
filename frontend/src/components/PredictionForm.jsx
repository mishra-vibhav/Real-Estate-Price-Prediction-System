import React, { useState } from 'react';
import './PredictionForm.css';

function PredictionForm({ onSubmit, loading, featureRanges }) {
  const [formData, setFormData] = useState({
    MedInc: 8.0,
    HouseAge: 20,
    AveRooms: 6.0,
    AveBedrms: 1.0,
    Population: 500,
    AveOccup: 2.5,
    Latitude: 37.8,
    Longitude: -122.2,
  });

  const features = [
    {
      key: 'MedInc',
      label: 'Median Income',
      tooltip: 'Median household income in block (in $100k units)',
    },
    {
      key: 'HouseAge',
      label: 'House Age',
      tooltip: 'Median house age in years',
    },
    {
      key: 'AveRooms',
      label: 'Avg Rooms per House',
      tooltip: 'Average number of rooms',
    },
    {
      key: 'AveBedrms',
      label: 'Avg Bedrooms per House',
      tooltip: 'Average number of bedrooms',
    },
    {
      key: 'Population',
      label: 'Population',
      tooltip: 'Block group population',
    },
    {
      key: 'AveOccup',
      label: 'Avg Occupancy',
      tooltip: 'Average occupancy ratio',
    },
    {
      key: 'Latitude',
      label: 'Latitude',
      tooltip: 'Block group latitude (California coordinates)',
    },
    {
      key: 'Longitude',
      label: 'Longitude',
      tooltip: 'Block group longitude (California coordinates)',
    },
  ];

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: parseFloat(value),
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(formData);
  };

  return (
    <div className="prediction-form">
      <h2>Enter Property Details</h2>
      <form onSubmit={handleSubmit}>
        <div className="form-grid">
          {features.map((feature) => {
            const range = featureRanges[feature.key];
            return (
              <div key={feature.key} className="form-group">
                <label htmlFor={feature.key}>
                  {feature.label}
                  <span className="tooltip-icon" title={feature.tooltip}>?</span>
                </label>
                <input
                  type="number"
                  id={feature.key}
                  name={feature.key}
                  value={formData[feature.key]}
                  onChange={handleChange}
                  step="0.1"
                  min={range?.min || 0}
                  max={range?.max || 100}
                  disabled={loading}
                />
                {range && (
                  <span className="range-hint">
                    {range.min} - {range.max}
                  </span>
                )}
              </div>
            );
          })}
        </div>

        <button
          type="submit"
          disabled={loading}
          className={`submit-btn ${loading ? 'loading' : ''}`}
        >
          {loading ? 'Predicting...' : 'Predict Price'}
        </button>
      </form>
    </div>
  );
}

export default PredictionForm;
