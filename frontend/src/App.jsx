import React, { useState } from "react";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [heatmap, setHeatmap] = useState(null);
  const [probabilities, setProbabilities] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const selected = e.target.files[0];
    if (selected) {
      setFile(selected);
      setPreview(URL.createObjectURL(selected));
      setResult(null);
      setHeatmap(null);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError("Please upload an image first");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch("http://127.0.0.1:10000/predict", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error("Prediction failed. Please try again.");
      }

      const data = await res.json();

      setResult(data);
      if (data.heatmap) {
        setHeatmap(`data:image/jpeg;base64,${data.heatmap}`);
      }
      setProbabilities(data.is_skin_lesion === false ? null : data.probabilities || null);
    } catch (err) {
      setError(err.message || "An error occurred. Please try again.");
      console.error("Error:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setHeatmap(null);
    setProbabilities(null);
    setError(null);
  };

  return (
    <div className="container">
      {/* Header */}
      <div className="header">
        <h1 className="header-title">DERMALENS</h1>
        <p className="header-subtitle">Advanced Skin Analysis AI</p>
      </div>

      {/* Upload Section */}
      <div className="upload-section">
        <div className="file-input-wrapper">
          <input
            type="file"
            id="fileInput"
            onChange={handleChange}
            accept="image/*"
          />
          <label htmlFor="fileInput" className="file-input-label">
            Choose Image
          </label>
        </div>
        {file && (
          <p className="no-file-message">
            Selected: <strong>{file.name}</strong>
          </p>
        )}
      </div>

      {/* Preview Section */}
      {preview && (
        <div className="preview-container">
          <div className="image-wrapper">
            <h3>Original Image</h3>
            <img src={preview} className="preview" alt="Original skin image" />
          </div>

          {/* Action Button */}
          <div style={{ display: "flex", flexDirection: "column", gap: "20px", justifyContent: "center" }}>
            <button
              className="predict-btn"
              onClick={handleUpload}
              disabled={loading}
            >
              {loading ? (
                <span>
                  Processing
                  <span className="loading"></span>
                  <span className="loading"></span>
                  <span className="loading"></span>
                </span>
              ) : (
                "Analyze Now"
              )}
            </button>

            {error && (
              <div
                style={{
                  padding: "15px",
                  background: "rgba(239, 68, 68, 0.1)",
                  border: "2px solid rgba(239, 68, 68, 0.4)",
                  borderRadius: "12px",
                  color: "#fca5a5",
                  fontSize: "14px",
                  textAlign: "center",
                }}
              >
                {error}
              </div>
            )}

            {(result || heatmap) && (
              <button
                className="predict-btn"
                onClick={handleClear}
                style={{
                  background: "linear-gradient(135deg, #64748b 0%, #475569 100%)",
                }}
              >
                New Analysis
              </button>
            )}
          </div>
        </div>
      )}

      {/* Results Section */}
      {result && result.is_skin_lesion === false && (
        <div className="result-section">
          <div
            className="result-card"
            style={{
              border: "2px solid rgba(249, 115, 22, 0.5)",
              background: "rgba(249, 115, 22, 0.08)",
            }}
          >
            <div className="result-card-content">
              <span className="prediction-label">Upload Rejected</span>
              <div className="prediction-result" style={{ color: "#f97316" }}>
                Not a skin lesion
              </div>
              <p style={{ color: "#94a3b8", fontSize: "13px", lineHeight: 1.6, marginTop: "14px" }}>
                {result.message}
              </p>
              {result.ood?.reason === "no_skin_tone" && (
                <p style={{ color: "#64748b", fontSize: "12px", marginTop: "10px" }}>
                  Skin-tone check failed: {((result.ood.skin_fraction || 0) * 100).toFixed(1)}% of
                  pixels look like skin (minimum {(result.ood.threshold || 0) * 100}%)
                </p>
              )}
              {result.ood && result.ood.score != null && (
                <p style={{ color: "#64748b", fontSize: "12px", marginTop: "10px" }}>
                  Out-of-distribution score {result.ood.score.toFixed(2)} vs
                  allowed {result.ood.threshold?.toFixed(2)}
                </p>
              )}
            </div>
          </div>
        </div>
      )}
      {result && result.is_skin_lesion !== false && (
        <div className="result-section">
          <div className="result-card">
            <div className="result-card-content">
              <span className="prediction-label">Analysis Result</span>
              <div className="prediction-result">
                {result.prediction}
              </div>

              <p style={{ color: "#94a3b8", fontSize: "13px", lineHeight: 1.6, marginTop: "14px" }}>
                This is an AI-assisted research/educational classifier, not a medical diagnosis.
              </p>

              <div className="confidence-section">
                <label className="confidence-label">Confidence Score</label>
                <div className="confidence-bar-container">
                  <div
                    className="confidence-bar"
                    style={{
                      width: `${result.confidence * 100}%`,
                    }}
                  ></div>
                </div>
                <div className="confidence-percentage">
                  {(result.confidence * 100).toFixed(1)}%
                </div>
              </div>
            </div>
          </div>

          {probabilities && (
            <div className="result-card probability-card">
              <div className="prediction-label">Class Probabilities</div>
              {Object.entries(probabilities).map(([label, value]) => (
                <div key={label} style={{ margin: "12px 0", textAlign: "left" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", gap: "12px", fontSize: "13px" }}>
                    <span>{label}</span><strong>{(value * 100).toFixed(1)}%</strong>
                  </div>
                  <div className="confidence-bar-container">
                    <div className="confidence-bar" style={{ width: `${value * 100}%` }} />
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Heatmap Section */}
          {heatmap && (
            <div className="image-wrapper">
              <div className="heatmap-title">
                <span className="heatmap-icon">🔥</span>
                Attention Map
              </div>
              <p className="heatmap-description">
                Shows areas the AI focused on for this prediction
              </p>
              <img
                src={heatmap}
                className="heatmap"
                alt="AI attention heatmap (Grad-CAM)"
              />
            </div>
          )}
        </div>
      )}

      {/* Footer Credit */}
      <div className="footer-credit">Developed by G - 11</div>
    </div>
  );
}

export default App;
