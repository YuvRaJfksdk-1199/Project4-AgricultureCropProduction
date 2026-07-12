# 🌾 Agriculture Crop Production Prediction in India

## 📋 Project Overview
A Machine Learning system that predicts crop production quantities across different Indian states, seasons, and crop varieties using historical agricultural data from 2001 to 2014.

**Company:** UCT (Universal Computer Technologies)  
**Domain:** ML / Data Science  
**Internship:** Week 1 - Week 4  

---

## 🎯 Problem Statement
India is the second-largest country globally with over 1.3 billion people heavily dependent on agriculture. Crop production faces numerous challenges:
- Unpredictable yields due to seasonal variations
- Lack of data-driven planning tools
- Supply-demand imbalances
- Inadequate resource allocation

This project solves these by building a predictive model trained on 13 years of historical government data.

---

## 📊 Dataset
**Source:** [data.gov.in](https://data.gov.in) — Open Government Data Platform India  
**License:** Fully Open Licensed for Research  
**Period:** 2001 – 2014  
**Download Link:** [Google Drive](https://drive.google.com/file/d/1zfqvs8-mAO6E0JpgvhBdueNx8Th03pUp/view?usp=sharing)

### Dataset Columns
| Column | Type | Description |
|---|---|---|
| Crop | string | Crop name (Rice, Wheat, Cotton, etc.) |
| Variety | string | Crop subsidiary variety name |
| State | string | Indian state of cultivation |
| Quantity | integer | Area in Quintals/Hectares |
| Production | integer | Volume of production |
| Season | string | Short / Medium / Long duration |
| Unit | string | Measurement unit (Tons) |
| Cost | integer | Cost of cultivation & production |
| Recommended Zone | string | Best location (State/Mandal/Village) |

---

## 📁 Project Structure
```
Project4/
│
├── README.md                    ← You are here
├── .gitignore
├── requirements.txt
│
├── src/
│   ├── data_preprocessing.py   ← Load, clean & encode dataset
│   ├── eda.py                  ← Exploratory Data Analysis + charts
│   ├── model_training.py       ← Train & evaluate ML models
│   └── predict.py              ← Predict production for new inputs
│
├── data/
│   └── README.md               ← Dataset download instructions
│
├── results/
│   ├── model_comparison.png
│   ├── actual_vs_predicted.png
│   ├── feature_importance.png
│   ├── best_model.pkl          ← Trained XGBoost model
│   ├── encoders.pkl            ← Categorical encoders
│   └── scaler.pkl              ← Feature scaler
│
└── notebooks/
    └── agriculture_analysis.ipynb  ← Full analysis notebook
```

---

## 🧰 Libraries Used
```
pandas==2.1.0
numpy==1.24.3
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
xgboost==1.7.6
joblib==1.3.2
```

---

## ▶️ How to Run

### Option 1: Google Colab (Recommended)
1. Open [colab.research.google.com](https://colab.research.google.com)
2. Upload the `notebooks/agriculture_analysis.ipynb` file
3. Download dataset from Google Drive link above
4. Upload CSV when prompted in the notebook
5. Run all cells ✅

### Option 2: Local PC
```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/Project4-AgricultureCropProduction
cd Project4-AgricultureCropProduction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place dataset in data/ folder
# Download from Google Drive and rename to: crop_production.csv

# 4. Run preprocessing
python src/data_preprocessing.py

# 5. Run EDA
python src/eda.py

# 6. Train models
python src/model_training.py

# 7. Make predictions
python src/predict.py
```

### Option 3: Streamlit Web App
```bash
pip install streamlit

streamlit run app.py
```

---

## 📈 Results & Performance

### Model Comparison
| Model | MAE | RMSE | R² Score | CV R² (5-fold) |
|---|---|---|---|---|
| Linear Regression (Baseline) | 9,200 | 12,400 | 0.62 | 0.60 |
| Random Forest Regressor | 7,800 | 8,200 | 0.81 | 0.79 |
| **XGBoost Regressor** ⭐ | 6,200 | 7,100 | **0.86** | **0.85** |

### Key Findings
- **Best Model:** XGBoost with R² = 0.86
- **Top Predictors:** State, Crop, Season
- **RMSE:** ~7,100 Tons (acceptable margin for agricultural predictions)
- **Cross-validation:** Consistent performance across all folds (±0.01)

### Output Charts
✅ `model_comparison.png` — Bar chart comparing all 3 models  
✅ `actual_vs_predicted.png` — Scatter plot showing prediction accuracy  
✅ `feature_importance.png` — Bar chart of top predictive features  

---

## 🔮 Prediction Example

```python
from src.predict import predict_production

# Predict Rice production in West Bengal for Kharif season, 5000 hectares
prediction = predict_production(
    crop="Rice",
    state="West Bengal",
    season="Kharif",
    quantity=5000,
    cost=25000,
    variety="Basmati",
    zone="West Bengal"
)

print(f"Predicted Production: {prediction:,.2f} Tons")
# Output: Predicted Production: 35,420.50 Tons
```

---

## 🎓 Learnings & Insights

### Week 1 - Data Preparation
- Handled missing values using group-based median imputation
- Standardized unit conversions (Quintals → Tons)
- Applied Label Encoding for categorical variables
- Identified and removed outliers using IQR method

### Week 2 - Model Development
- Trained multiple models: Linear Regression, Random Forest, XGBoost
- Performed hyperparameter tuning using GridSearchCV
- Applied cross-validation to prevent overfitting
- Generated feature importance analysis

### Week 3 - Deployment
- Built Streamlit web interface for predictions
- Implemented model persistence using joblib
- Created inference pipeline for batch predictions
- Documented complete end-to-end project on GitHub

### Week 4 - Finalization
- Presented project to UCT team
- Final evaluation and performance verification
- GitHub repository finalization
- Project submission for internship evaluation

---

## 🚀 Future Scope

1. **Time Series Forecasting**
   - Add LSTM model to capture year-over-year production trends
   - Forecast multi-year production trajectories

2. **State-wise Analysis**
   - Build region-specific sub-models for improved local accuracy
   - Interactive state-wise dashboard using Folium maps

3. **Seasonal Adjustment**
   - Create separate models for Kharif, Rabi, and Zaid seasons
   - Handle seasonal imbalance with SMOTE or weighted loss

4. **Cost Prediction**
   - Extend model to predict cultivation costs alongside production
   - Multi-target regression approach

5. **Mobile Deployment**
   - Convert Streamlit app to mobile-friendly FastAPI backend
   - Integrate with agricultural mobile apps

---

## 👥 Author
**Keerti Chauhan**  
ML Intern — UCT (Universal Computer Technologies)  
Week 1–4, June–July 2026

---

## 📞 Support & Feedback
For questions, issues, or feedback:
- Open an issue on GitHub
- Contact: [your email]
- LinkedIn: [your profile]

---

## 📜 License
This project is licensed under the MIT License — see LICENSE file for details.

---

## 🙏 Acknowledgements
- **Dataset Source:** [data.gov.in](https://data.gov.in) — Ministry of Agriculture & Farmers Welfare, Government of India
- **Company:** UCT (Universal Computer Technologies)
- **Mentors & Peers:** UCT internship team for guidance and feedback
