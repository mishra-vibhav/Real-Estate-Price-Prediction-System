import React, { useState, useEffect } from 'react';
import axios from 'axios';
import toast, { Toaster } from 'react-hot-toast';
import PredictionForm from './components/PredictionForm';
import ResultsDisplay from './components/ResultsDisplay';
import ModelStats from './components/ModelStats';
import './App.css';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function App() {
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [featureRanges, setFeatureRanges] = useState(null);

  useEffect(() => {
    const fetchModelInfo = async () => {
      try {
        const [infoRes, rangesRes] = await Promise.all([
          axios.get(`${API_BASE}/model-info`),
          axios.get(`${API_BASE}/feature-ranges`)
        ]);
        setModelInfo(infoRes.data);
        setFeatureRanges(rangesRes.data);
      } catch (error) {
        toast.error('Failed to load model information');
        console.error(error);
      }
    };

    fetchModelInfo();
  }, []);

  const handlePredict = async (formData) => {
    setLoading(true);
    try {
      const response = await axios.post(`${API_BASE}/predict`, formData);
      setPrediction(response.data);
      toast.success('Prediction generated successfully!');
    } catch (error) {
      toast.error('Failed to generate prediction');
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <Toaster position="top-right" />
      
      <header className="app-header">
        <div className="header-content">
          <h1>🏠 Real Estate Price Predictor</h1>
          <p>ML-Powered Property Valuation System</p>
        </div>
      </header>

      <main className="app-main">
        <div className="container">
          <div className="layout">
            <div className="left-panel">
              {modelInfo && <ModelStats data={modelInfo} />}
            </div>

            <div className="center-panel">
              {featureRanges && (
                <PredictionForm 
                  onSubmit={handlePredict}
                  loading={loading}
                  featureRanges={featureRanges}
                />
              )}
            </div>

            <div className="right-panel">
              {prediction && <ResultsDisplay prediction={prediction} />}
            </div>
          </div>
        </div>
      </main>

      <footer className="app-footer">
        <p>Powered by XGBoost & California Housing Dataset</p>
      </footer>
    </div>
  );
}

export default App;
