import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import './ModelStats.css';

function ModelStats({ data }) {
  const statsData = [
    { name: 'R² Score', value: (data.r2_score * 100).toFixed(2), unit: '%' },
    { name: 'RMSE', value: (data.rmse / 1000).toFixed(1), unit: 'K' },
    { name: 'MAE', value: (data.mae / 1000).toFixed(1), unit: 'K' },
  ];

  const chartData = [
    { metric: 'R² Score', value: data.r2_score * 100 },
    { metric: 'CV Mean', value: data.cv_r2_mean * 100 },
  ];

  return (
    <div className="model-stats">
      <h2>Model Performance</h2>
      
      <div className="stats-grid">
        {statsData.map((stat, idx) => (
          <div key={idx} className="stat-card">
            <div className="stat-label">{stat.name}</div>
            <div className="stat-value">
              {stat.value} <span className="stat-unit">{stat.unit}</span>
            </div>
          </div>
        ))}
      </div>

      <div className="chart-container">
        <h3>Accuracy Metrics</h3>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="metric" />
            <YAxis />
            <Tooltip formatter={(value) => `${value.toFixed(2)}%`} />
            <Bar dataKey="value" fill="#3b82f6" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="training-info">
        <h3>Training Details</h3>
        <p><strong>Samples Used:</strong> {data.training_samples.toLocaleString()}</p>
        <p><strong>Features:</strong> {data.feature_count || data.features.length}</p>
        <p><strong>Model Type:</strong> {data.model_type}</p>
      </div>
    </div>
  );
}

export default ModelStats;
