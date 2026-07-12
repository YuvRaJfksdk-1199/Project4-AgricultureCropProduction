"""
Project 4: Agriculture Crop Production Prediction
File: src/predict.py
Description: Predict crop production for new input data
Company: UCT — Machine Learning Internship
"""

import numpy as np
import pandas as pd
import joblib
import os

RESULTS_DIR = "results"

def predict_production(crop, state, season, quantity, cost, variety="Unknown", zone="Unknown"):
    """
    Predict crop production for given inputs.

    Parameters:
        crop     : str  — Crop name (e.g., "Rice", "Wheat")
        state    : str  — Indian state (e.g., "Punjab", "Maharashtra")
        season   : str  — Season (e.g., "Kharif", "Rabi")
        quantity : float — Area under cultivation (Hectares)
        cost     : float — Cost of cultivation (INR)
        variety  : str  — Crop variety (optional)
        zone     : str  — Recommended zone (optional)

    Returns:
        float — Predicted production in Tons
    """

    # Load saved model and preprocessors
    model_path = f"{RESULTS_DIR}/best_model.pkl"
    encoders_path = f"{RESULTS_DIR}/encoders.pkl"
    scaler_path = f"{RESULTS_DIR}/scaler.pkl"

    if not os.path.exists(model_path):
        print("❌ Model not found! Please run model_training.py first.")
        return None

    model = joblib.load(model_path)
    encoders = joblib.load(encoders_path)
    scaler = joblib.load(scaler_path)

    # Encode categorical inputs
    def safe_encode(encoder, value):
        try:
            return encoder.transform([value])[0]
        except ValueError:
            return 0

    crop_enc = safe_encode(encoders.get('crop', None), crop) if 'crop' in encoders else 0
    variety_enc = safe_encode(encoders.get('variety', None), variety) if 'variety' in encoders else 0
    state_enc = safe_encode(encoders.get('state', None), state) if 'state' in encoders else 0
    season_enc = safe_encode(encoders.get('season', None), season) if 'season' in encoders else 0
    zone_enc = safe_encode(encoders.get('recommended_zone', None), zone) if 'recommended_zone' in encoders else 0

    # Derived features
    cost_per_unit = cost / (quantity + 1)
    production_efficiency = 1.0

    # Scale numerical features (DataFrame keeps column names, avoids sklearn warnings)
    numerical_df = pd.DataFrame(
        [[quantity, cost, cost_per_unit, production_efficiency]],
        columns=['quantity', 'cost', 'cost_per_unit', 'production_efficiency']
    )
    numerical_scaled = scaler.transform(numerical_df)[0]

    # Build feature vector as a DataFrame matching training column order/names
    feature_cols = [
        'crop_encoded', 'variety_encoded', 'state_encoded', 'season_encoded',
        'quantity', 'cost', 'cost_per_unit', 'production_efficiency',
        'recommended_zone_encoded'
    ]
    features = pd.DataFrame([[
        crop_enc, variety_enc, state_enc, season_enc,
        numerical_scaled[0], numerical_scaled[1],
        numerical_scaled[2], numerical_scaled[3], zone_enc
    ]], columns=feature_cols)

    prediction = model.predict(features)[0]
    return max(0, prediction)


# ── TEST ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  CROP PRODUCTION PREDICTOR")
    print("  UCT — Machine Learning Internship")
    print("=" * 55)

    test_cases = [
        {
            "crop": "Rice", "state": "West Bengal",
            "season": "Kharif", "quantity": 5000, "cost": 25000,
            "variety": "Basmati", "zone": "West Bengal"
        },
        {
            "crop": "Wheat", "state": "Punjab",
            "season": "Rabi", "quantity": 8000, "cost": 18000,
            "variety": "HD-2967", "zone": "Punjab"
        },
        {
            "crop": "Cotton", "state": "Maharashtra",
            "season": "Kharif", "quantity": 3500, "cost": 32000,
            "variety": "BT Cotton", "zone": "Maharashtra"
        },
    ]

    print(f"\n{'Crop':<10} {'State':<15} {'Season':<8} {'Quantity':>10} {'Predicted (Tons)':>20}")
    print("-" * 75)

    for tc in test_cases:
        pred = predict_production(**tc)
        if pred is not None:
            print(f"{tc['crop']:<10} {tc['state']:<15} {tc['season']:<8} "
                  f"{tc['quantity']:>10,} {pred:>18,.2f}")

    print("\n✅ Predictions Complete!")
