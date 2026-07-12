# Dataset — Agriculture Crop Production India

## Download Instructions

The dataset is **NOT included** in this repository due to size constraints. Download it separately from:

👉 **[Google Drive Link](https://drive.google.com/file/d/1zfqvs8-mAO6E0JpgvhBdueNx8Th03pUp/view?usp=sharing)**

### Steps:
1. Click the Google Drive link above
2. Click **Download** button (or right-click → Save)
3. Rename the file to: `crop_production.csv`
4. Place it in this `data/` folder

```
Project4/
└── data/
    ├── README.md
    └── crop_production.csv  ← Place downloaded file here
```

---

## Dataset Information

**Source:** data.gov.in — Open Government Data Platform India  
**License:** Fully Open Licensed for Research Use  
**Period:** 2001 – 2014  
**Size:** ~20,000 records  

---

## Columns Description

| Column | Type | Description | Example |
|---|---|---|---|
| Crop | string | Name of the crop | Rice, Wheat, Cotton |
| Variety | string | Crop subsidiary variety | Basmati, HD-2967, BT Cotton |
| State | string | Indian state of cultivation | Punjab, West Bengal, Maharashtra |
| Quantity | integer | Area under cultivation | 5000 (Hectares) |
| Production | integer | Volume of production | 35000 (Tons) |
| Season | string | Duration category | Kharif, Rabi, Zaid |
| Unit | string | Measurement unit | Tons, Quintals |
| Cost | integer | Cost of cultivation & production | 25000 (INR) |
| Recommended Zone | string | Best suited location | State/Mandal/Village |

---

## Data Quality Notes

- Missing values are handled via median/mode imputation during preprocessing
- Unit conversions (Quintals → Tons) are applied automatically
- Outliers in production quantities are detected and removed using IQR method
- Categorical variables are Label Encoded for model training

---

## Usage

Once the CSV file is placed in this folder, run:

```bash
python ../src/data_preprocessing.py
```

This will automatically:
- Load the CSV
- Clean missing values
- Encode categorical variables
- Scale numerical features
- Split into train/test sets

No manual data preparation needed! ✅
