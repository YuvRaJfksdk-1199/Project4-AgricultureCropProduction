# Results

This folder contains all outputs from model training, evaluation, and inference.

## Generated Files

### Model Files
- `best_model.pkl` — Trained XGBoost model (final production model)
- `encoders.pkl` — LabelEncoder objects for categorical features
- `scaler.pkl` — MinMaxScaler for numerical features

### Visualization Charts
- `chart1_top_states.png` — Top 10 states by production
- `chart2_top_crops.png` — Top 10 crops by production
- `chart3_season_distribution.png` — Pie chart of seasonal distribution
- `chart4_correlation_heatmap.png` — Feature correlation heatmap
- `chart5_production_distribution.png` — Histogram of production values

### Model Evaluation
- `model_comparison.png` — Bar chart comparing all 3 models (Linear Regression, Random Forest, XGBoost)
- `actual_vs_predicted.png` — Scatter plot: actual vs. predicted production values
- `feature_importance.png` — Bar chart of top predictive features

## Model Performance

**Final Model:** XGBoost Regressor

| Metric | Value |
|---|---|
| R² Score | 0.86 |
| RMSE | ~7,100 Tons |
| MAE | ~6,200 Tons |
| Cross-validation R² | 0.85 ± 0.01 |

## How to Use the Saved Model

```python
import joblib

# Load the model and preprocessors
model = joblib.load('best_model.pkl')
encoders = joblib.load('encoders.pkl')
scaler = joblib.load('scaler.pkl')

# Make predictions
import numpy as np
features = np.array([[...]])  # Encoded and scaled features
prediction = model.predict(features)
```

Or use the prediction script:
```bash
python ../src/predict.py
```

## Generating Results

To regenerate all charts and models:

```bash
# Run preprocessing
python ../src/data_preprocessing.py

# Run EDA (generates charts 1-5)
python ../src/eda.py

# Train models (generates comparison charts and model files)
python ../src/model_training.py
```

All outputs will be saved to this `results/` folder automatically.
