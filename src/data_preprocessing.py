"""
Project 4: Agriculture Crop Production Prediction
File: src/data_preprocessing.py
Description: Load, clean, encode and split the dataset
Company: UCT — Machine Learning Internship
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

# ── CONFIG ──────────────────────────────────────────────────
DATA_PATH = "data/crop_production.csv"
RANDOM_STATE = 42

# ── STEP 1: LOAD DATA ───────────────────────────────────────
def load_data(path=DATA_PATH):
    """Load the crop production CSV dataset."""
    print("=" * 50)
    print("STEP 1: Loading Dataset")
    print("=" * 50)

    df = pd.read_csv(path)
    print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    return df


# ── STEP 2: CLEAN DATA ──────────────────────────────────────
def clean_data(df):
    """Handle missing values, fix units, remove duplicates."""
    print("\n" + "=" * 50)
    print("STEP 2: Cleaning Data")
    print("=" * 50)

    # Drop duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"✅ Removed {before - len(df)} duplicate rows")

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Fill missing numerical values with median
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
            print(f"✅ Filled missing values in '{col}'")

    # Fill missing categorical values with mode
    cat_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
            print(f"✅ Filled missing values in '{col}'")

    # Remove outliers using IQR on production
    if 'production' in df.columns:
        Q1 = df['production'].quantile(0.25)
        Q3 = df['production'].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        before = len(df)
        df = df[(df['production'] >= lower) & (df['production'] <= upper)]
        print(f"✅ Removed {before - len(df)} outlier rows")

    print(f"\n✅ Cleaned dataset shape: {df.shape}")
    return df


# ── STEP 3: FEATURE ENGINEERING ─────────────────────────────
def feature_engineering(df):
    """Create new meaningful features."""
    print("\n" + "=" * 50)
    print("STEP 3: Feature Engineering")
    print("=" * 50)

    if 'cost' in df.columns and 'quantity' in df.columns:
        df['cost_per_unit'] = df['cost'] / (df['quantity'] + 1)
        print("✅ Created feature: cost_per_unit")

    if 'production' in df.columns and 'quantity' in df.columns:
        df['production_efficiency'] = df['production'] / (df['quantity'] + 1)
        print("✅ Created feature: production_efficiency")

    return df


# ── STEP 4: ENCODE CATEGORICAL VARIABLES ────────────────────
def encode_features(df):
    """Label encode categorical columns."""
    print("\n" + "=" * 50)
    print("STEP 4: Encoding Categorical Features")
    print("=" * 50)

    cat_cols = ['crop', 'variety', 'state', 'season', 'unit', 'recommended_zone']
    encoders = {}

    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            print(f"✅ Encoded '{col}' → '{col}_encoded'")

    return df, encoders


# ── STEP 5: SCALE NUMERICAL FEATURES ────────────────────────
def scale_features(df):
    """Normalize numerical features to [0, 1] range."""
    print("\n" + "=" * 50)
    print("STEP 5: Scaling Numerical Features")
    print("=" * 50)

    scale_cols = ['quantity', 'cost', 'cost_per_unit', 'production_efficiency']
    scaler = MinMaxScaler()
    existing_cols = [c for c in scale_cols if c in df.columns]

    df[existing_cols] = scaler.fit_transform(df[existing_cols])
    print(f"✅ Scaled columns: {existing_cols}")
    return df, scaler


# ── STEP 6: PREPARE FEATURES & TARGET ───────────────────────
def prepare_xy(df):
    """Select feature columns and target variable."""
    print("\n" + "=" * 50)
    print("STEP 6: Preparing Features & Target")
    print("=" * 50)

    feature_cols = [
        'crop_encoded', 'variety_encoded', 'state_encoded',
        'season_encoded', 'quantity', 'cost', 'cost_per_unit',
        'production_efficiency', 'recommended_zone_encoded'
    ]

    feature_cols = [c for c in feature_cols if c in df.columns]
    target_col = 'production'

    X = df[feature_cols]
    y = df[target_col]

    print(f"✅ Features ({len(feature_cols)}): {feature_cols}")
    print(f"✅ Target: '{target_col}'")
    return X, y


# ── STEP 7: TRAIN-TEST SPLIT ─────────────────────────────────
def split_data(X, y):
    """Split into 80% train and 20% test."""
    print("\n" + "=" * 50)
    print("STEP 7: Train-Test Split (80/20)")
    print("=" * 50)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    print(f"✅ Training set: {X_train.shape[0]} samples")
    print(f"✅ Testing set:  {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test


# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    df = feature_engineering(df)
    df, encoders = encode_features(df)
    df, scaler = scale_features(df)
    X, y = prepare_xy(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("\n✅ Preprocessing Complete! Ready for model training.")
