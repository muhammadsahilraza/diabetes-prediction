# 🩸 Diabetes Prediction — End-to-End Machine Learning Project

**Developed by Muhammad Sahil Raza** | Machine Learning Engineer

An end-to-end Machine Learning project that predicts whether a patient is at risk of diabetes using the **Pima Indians Diabetes Dataset**. This project covers data cleaning, exploratory data analysis (EDA), multi-model comparison, hyperparameter tuning with GridSearchCV, and a **Streamlit** web application for live predictions.

---

## 📂 Project Structure

```text
diabetes-prediction/
│
├── Diabetes_Prediction.ipynb    # Full ML pipeline (EDA → Cleaning → Training → Tuning → Evaluation)
├── app.py                       # Streamlit web application
├── requirements.txt             # Python dependencies
├── model.pkl                    # Tuned XGBoost model (best performer)
├── scaler.pkl                   # Fitted StandardScaler
├── Dataset/
│   └── diabetes.csv             # Pima Indians Diabetes Dataset
├── .gitignore                   # Git ignore rules
└── README.md
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/MuhammadSahilRaza/diabetes-prediction.git
cd diabetes-prediction

# Create virtual environment (recommended)
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 How to Run

### 1. Train the Model

```bash
jupyter notebook Diabetes_Prediction.ipynb
```

Run all cells top-to-bottom. This generates:

- `model.pkl` — tuned XGBoost model
- `scaler.pkl` — fitted StandardScaler

### 2. Launch the Web App

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 📊 Dataset Overview

**Pima Indians Diabetes Dataset** — 768 rows, 8 medical features, 1 binary target.

| Feature | Description |
|---|---|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration (mg/dL) |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-Hour serum insulin (mu U/ml) |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Genetic diabetes risk score |
| Age | Age in years |
| **Outcome** | 0 = No Diabetes, 1 = Diabetes |

---

## 🧠 Models Compared

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting
5. **XGBoost** ✅
6. Support Vector Machine (SVM)
7. K-Nearest Neighbors (KNN)

---

## 📈 Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| **XGBoost (tuned)** 🏆 | **0.766** | 0.680 | 0.630 | 0.654 | 0.820 |
| Gradient Boosting | 0.760 | 0.689 | 0.574 | 0.626 | **0.831** |
| KNN | 0.753 | 0.660 | 0.611 | 0.635 | 0.789 |
| Random Forest | 0.740 | 0.652 | 0.556 | 0.600 | 0.816 |
| SVM | 0.740 | 0.652 | 0.556 | 0.600 | 0.796 |
| Logistic Regression | 0.708 | 0.600 | 0.500 | 0.545 | 0.813 |
| Decision Tree | 0.682 | 0.553 | 0.481 | 0.515 | 0.636 |

### 🏆 Best Model: XGBoost (Tuned via GridSearchCV)

**Selected Hyperparameters (5-fold Cross-Validation):**

```python
{
    'learning_rate': 0.01,
    'max_depth': 3,
    'n_estimators': 300,
    'subsample': 0.8
}
```

- **Cross-Validation Score:** 77.20%
- **Test Accuracy:** 76.62%
- **ROC-AUC:** 0.820

The gap between CV (77.2%) and Test (76.6%) is **only 0.6%** — a strong signal that the model is **not overfitting**.

---

## ❓ Why Is Accuracy ~76% and Not 90%+?

### 1. Small Dataset (768 rows only)

With only 768 samples and 8 features, the model has limited data to learn complex patterns. Achieving 90%+ accuracy typically requires **10,000+ rows** with richer features.

### 2. Medical Data Is Inherently Noisy

Blood glucose, insulin, and BMI carry **biological variance** that 8 numeric features cannot fully capture.

### 3. Class Imbalance (65% / 35%)

500 non-diabetic vs 268 diabetic samples. This limits achievable accuracy without sacrificing recall.

### 4. 90%+ Claims Are Red Flags

- **Data leakage** — scaling before train-test split
- **Overfitting** — tuning on test data
- **Duplicate rows** not removed
- **Fabricated metrics**

### 5. Published Benchmarks Confirm This

Research (Smith et al., 1988) and Kaggle leaderboards report **76–82% accuracy** as the realistic ceiling.

> "An honest 76% with a leak-free pipeline is worth far more than a fake 95%."

---

## 🎯 Why Recall & ROC-AUC Matter More Than Accuracy

| Error Type | Meaning | Real-World Cost |
|---|---|---|
| **False Negative** | Diabetic patient labeled healthy | 🔴 Dangerous — no treatment |
| **False Positive** | Healthy person flagged for testing | 🟢 Low — extra tests only |

- **Recall (0.63)** — correctly identifying diabetic patients
- **ROC-AUC (0.820)** — model's ability to separate classes

---

## 🛠️ Preprocessing Pipeline

1. **Zero-value handling** — Replaced invalid `0` values with `NaN`.
2. **Median imputation** — Filled NaNs using median.
3. **Train-test split** — Stratified 80/20 split.
4. **Feature scaling** — `StandardScaler` fit on train only.
5. **Hyperparameter tuning** — `GridSearchCV` with 5-fold CV.

---

## 🌐 Deployment

1. Push the repository to GitHub.
2. Visit **https://share.streamlit.io**
3. Click **"New app"** → connect GitHub repo.
4. Set **Main file path:** `app.py`
5. Click **Deploy** ✅

---

## 🔮 Future Improvements

- Use larger datasets like **CDC Diabetes Health Indicators (253K rows)** → 85–88% achievable.
- Add domain feature engineering.
- Apply **SMOTE / ADASYN** for class balancing.
- Build a **Stacking / Voting Ensemble**.
- Add **SHAP** explainability.
- Deploy with **FastAPI + Docker**.

---

## 🧾 Tech Stack

- **Language:** Python 3.10+
- **Data:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, XGBoost
- **Visualization:** Matplotlib, Seaborn
- **Deployment:** Streamlit
- **Model Persistence:** Joblib

---

## 👤 Author

**Muhammad Sahil Raza**

- GitHub: [@MuhammadSahilRaza](https://github.com/muhammadsahilraza)
- LinkedIn: [muhammad-sahil-raza](https://www.linkedin.com/in/muhammad-sahil-raza-3a1888326/)

---

## 📜 License

This project is licensed under the **MIT License** — free to use, modify, and distribute.

---

## ⚠️ Disclaimer

This tool is built for **educational purposes only** and is not a substitute for professional medical advice.

---

<p align="center">
  <strong>⭐ If you found this project useful, please consider giving it a star on GitHub!</strong>
</p>