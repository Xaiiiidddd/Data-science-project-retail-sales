# Data-science-project-retail-sales 
# 🛒 Retail Sales Analytics — End-to-End Data Science Project

A complete data science project on **retail sales data** covering exploratory data analysis, machine learning, and customer segmentation using Python.

---

## 📁 Project Structure

```
retail_ds_project/
│
├── data/
│   └── generate_data.py          # Synthetic dataset generator
│
├── src/
│   ├── eda.py                    # Exploratory Data Analysis
│   ├── feature_engineering.py    # Feature creation for ML
│   ├── model_training.py         # ML model training & evaluation
│   ├── rfm_segmentation.py       # RFM customer segmentation
│   └── visualizations.py         # All plots & charts
│
├── notebooks/
│   └── retail_analysis.ipynb     # Full Jupyter walkthrough
│
├── outputs/                      # Auto-generated charts (PNG)
├── models/                       # Saved trained models (.pkl)
│
├── main.py                       # Run entire pipeline end-to-end
├── requirements.txt
└── README.md
```

---

## 🎯 Project Goals

| Goal | Method |
|------|--------|
| Understand sales trends | Time-series EDA |
| Identify top categories & regions | GroupBy aggregations |
| Predict Customer Lifetime Value | Regression ML models |
| Segment customers for targeting | RFM Analysis |
| Measure discount effectiveness | Correlation analysis |

---

## 📊 Dataset

Synthetic retail dataset with **5,000 transactions** across **500 customers** (2022–2024).

| Column | Description |
|--------|-------------|
| `transaction_id` | Unique transaction ID |
| `customer_id` | Unique customer ID |
| `date` | Transaction date |
| `category` | Product category (7 types) |
| `quantity` | Units purchased |
| `unit_price` | Price per unit ($) |
| `discount` | Discount applied (0–20%) |
| `revenue` | Net revenue after discount |
| `age` | Customer age |
| `region` | Geographic region |
| `segment` | Customer tier (Premium/Regular/Budget) |
| `loyalty_years` | Years as a customer |

---

## 🤖 Machine Learning — CLV Prediction

Three models trained to predict **Customer Lifetime Value**:

| Model | R² (test) | MAE | CV R² |
|-------|-----------|-----|-------|
| Linear Regression | 0.949 | $539 | 0.919 |
| Random Forest | 0.967 | $375 | 0.949 |
| **Gradient Boosting** ✅ | **0.982** | **$315** | **0.977** |

Top predictors: **max single order value** (57.4%), order frequency (20%), avg order value (18.6%).

---

## 👥 RFM Customer Segments

| Segment | Count | Action |
|---------|-------|--------|
| 🏆 Champions | 121 | Reward, upsell, ask for referrals |
| 💙 Loyal | 133 | Loyalty programs, early access |
| ⚠️ At Risk | 128 | Win-back campaigns, discounts |
| 😴 Lost | 118 | Bold reactivation or write off |

---

## 💡 Key Business Insights

1. **Revenue recovered +17.8%** in 2024 after a 2023 dip
2. **Electronics = 44% of revenue** at $1,815 avg order value
3. **Central region** leads; **East region** is undertapped
4. **Discounts reduce revenue**: every 5% off cuts avg order by ~$50
5. **49.2% of customers** are At Risk or Lost — urgent retention needed
6. Budget-segment customers spend ~$200 more annually than Premium

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/retail-ds-project.git
cd retail-ds-project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline
python main.py
```

Charts are saved to `outputs/`, trained models to `models/`.

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **pandas** — data manipulation
- **numpy** — numerical computing
- **scikit-learn** — ML models, cross-validation, metrics
- **matplotlib / seaborn** — static visualizations
- **joblib** — model persistence

---

## 📈 Sample Outputs

Running `main.py` generates these charts in `outputs/`:

- `monthly_revenue_trend.png`
- `category_performance.png`
- `regional_sales.png`
- `weekday_sales.png`
- `discount_impact.png`
- `feature_importance.png`
- `actual_vs_predicted.png`
- `rfm_segments.png`
- `model_comparison.png`

---

## 📄 License

MIT License — free to use, modify, and distribute.
