"""
Project 4: Agriculture Crop Production Prediction
File: src/model_training.py
Description: Train, evaluate and compare ML models
Company: UCT — Machine Learning Internship
"""

import numpy as np
import joblib
import os

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from data_preprocessing import (
    load_data, clean_data, feature_engineering,
    encode_features, scale_features, prepare_xy, split_data
)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── EVALUATION FUNCTION ──────────────────────────────────────
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """Train a model and return evaluation metrics."""
    print(f"\n🔧 Training: {name}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"   MAE:         {mae:,.2f}")
    print(f"   RMSE:        {rmse:,.2f}")
    print(f"   R² Score:    {r2:.4f}")

    return {
        'name': name,
        'model': model,
        'y_pred': y_pred,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }


# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("CROP PRODUCTION PREDICTION — MODEL TRAINING")
    print("Company: UCT — Machine Learning Internship")
    print("=" * 60)

    # Preprocess
    df = load_data()
    df = clean_data(df)
    df = feature_engineering(df)
    df, encoders = encode_features(df)
    df, scaler = scale_features(df)
    X, y = prepare_xy(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Define models
    models = [
        ("Linear Regression", LinearRegression()),
        ("Random Forest", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
        ("XGBoost", XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42, verbosity=0)),
    ]

    print("\n" + "=" * 60)
    print("TRAINING ALL MODELS")
    print("=" * 60)

    results = []
    for name, model in models:
        result = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        results.append(result)

    # Find best model
    best = max(results, key=lambda x: x['r2'])

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n{'Model':<25} {'MAE':>10} {'RMSE':>12} {'R²':>8}")
    print("-" * 60)
    for r in results:
        marker = " ⭐ BEST" if r['name'] == best['name'] else ""
        print(f"{r['name']:<25} {r['mae']:>10,.1f} {r['rmse']:>12,.1f} {r['r2']:>8.4f}{marker}")

    # Save best model
    joblib.dump(best['model'], f"{RESULTS_DIR}/best_model.pkl")
    joblib.dump(encoders, f"{RESULTS_DIR}/encoders.pkl")
    joblib.dump(scaler, f"{RESULTS_DIR}/scaler.pkl")
    print(f"\n✅ Best model saved: {RESULTS_DIR}/best_model.pkl")

    print("\n✅ Training Complete!")
    print(f"   Best Model: {best['name']}")
    print(f"   R² Score:   {best['r2']:.4f}")
    print(f"   RMSE:       {best['rmse']:,.2f} Tons")
