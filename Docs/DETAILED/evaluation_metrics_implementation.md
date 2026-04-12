# Autoencoder Evaluation Metrics Implementation

## Overview
A comprehensive evaluation pipeline was implemented to assess the performance of the Autoencoder models (Dense & LSTM) used for anomaly detection in the Zynaptrix Industrial Copilot. This pipeline calculates 8 key metrics, generates visual plots, and provides API endpoints for data retrieval.

## Automatically Evaluating New Machines
The evaluation pipeline is now fully integrated into the machine enrollment flow. 
When a new machine is added via the platform, the backend automatically triggers a sequence of operations:
1. Dataset generation containing normal and fault patterns (`generate_dataset.py`)
2. Normalization of sensor data (`preprocessing/normalization.py`)
3. Training of the Autoencoder models (`models/train_model.py`)
4. **NEW: Evaluation of the trained models (`models/evaluate_model.py`)**

✅ **Automatic Evaluation**: You do *not* need to manually run evaluation scripts when adding new machines. The system will automatically compute the metrics and generate the plots exactly as it did for the initial 4 machines.

## Implemented Metrics
The following 8 metrics are calculated using labeled data where states like `machine_fault`, `sensor_drift`, `sensor_freeze`, and `idle` are mapped to anomalous labels, while `normal` is the primary non-anomalous class.

| Metric | What It Measures |
| --- | --- |
| **Precision** | Accuracy of flagged anomalies |
| **Recall / Sensitivity** | Percentage of real faults that were successfully caught |
| **F1 Score** | Harmonic mean of precision & recall |
| **AUC-ROC** | Discrimination ability across thresholds |
| **False Positive Rate (FPR)** | Sensor glitches incorrectly flagged as anomalies |
| **False Negative Rate (FNR)** | Missed real faults (critical for safety) |
| **Threshold Selection** | Visualization of F1, Precision, and Recall across thresholds to justify the mean + 2σ boundary |
| **MSE Distribution** | Histogram visualizing the overlapping boundary between normal and fault conditions |

## New Core Files
- `models/evaluate_model.py`: Core evaluation engine that loads the model, evaluates the dataset, computes metrics, and generates 4 publication-quality plots.
- `api/evaluation_routes.py`: FastAPI routes to fetch JSON evaluation metrics and specific PNG plot visualisations.

## API Endpoints
- `GET /api/evaluation/summary`: Retrieves a comparison summary across all evaluated machines.
- `GET /api/evaluation/{machine_id}`: Retrieves the detailed metrics JSON for a specific machine.
- `GET /api/evaluation/{machine_id}/plots/{plot_name}`: Serves the specific plot image file (e.g., `roc_curve_dense.png`, `mse_distribution_dense.png`).

## Stored Outputs
Outputs are saved to `backend/data/processed/evaluation/{machine_id}/`:
- `metrics_{model_type}.json` (Detailed mathematical readouts)
- `roc_curve_{model_type}.png`
- `mse_distribution_{model_type}.png`
- `confusion_matrix_{model_type}.png`
- `threshold_sweep_{model_type}.png`
- `evaluation_summary.json` (Root level, compares all models)
